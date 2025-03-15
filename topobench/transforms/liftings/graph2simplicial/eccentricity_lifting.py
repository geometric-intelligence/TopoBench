"""This module implements the SimplicialEccentricityLifting class, which lifts graphs to simplicial complexes."""

from itertools import combinations

import networkx as nx
import torch_geometric
from toponetx.classes import SimplicialComplex

from topobench.transforms.liftings.graph2simplicial.base import (
    Graph2SimplicialLifting,
)


class SimplicialEccentricityLifting(Graph2SimplicialLifting):
    r"""Lift graphs to simplicial complex domain using eccentricity.

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
        simplicial_complex = SimplicialComplex()
        eccentricities = nx.eccentricity(graph)
        simplices = [set() for _ in range(2, self.complex_dim + 1)]

        for node in graph.nodes:
            simplicial_complex.add_node(node, features=data.x[node])

        for node, ecc in eccentricities.items():
            neighborhood = list(
                nx.single_source_shortest_path_length(
                    graph, node, cutoff=ecc
                ).keys()
            )
            for k in range(1, self.complex_dim):
                for combination in combinations(neighborhood, k + 1):
                    simplices[k - 1].add(tuple(sorted(combination)))

        for set_k_simplices in simplices:
            simplicial_complex.add_simplices_from(list(set_k_simplices))

        return self._get_lifted_topology(simplicial_complex, graph)
