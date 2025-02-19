from itertools import combinations

import networkx as nx
import torch_geometric
from toponetx.classes import SimplicialComplex

from topobenchmark.transforms.liftings.graph2simplicial.base import (
    Graph2SimplicialLifting,
)


class SimplicialVietorisRipsLifting(Graph2SimplicialLifting):
    r"""Lifts graphs to simplicial complex domain using the Vietoris-Rips complex based on pairwise distances.

    Parameters
    ----------
    distance_threshold : float
        The maximum distance between vertices to form a simplex.
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, distance_threshold=1.0, **kwargs):
        super().__init__(**kwargs)
        self.distance_threshold = distance_threshold

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Lifts the topology of a graph to a simplicial complex using the Vietoris-Rips complex.

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

        # Calculate pairwise shortest path distances
        path_lengths = dict(nx.all_pairs_shortest_path_length(graph))

        for k in range(2, self.complex_dim + 1):
            for combination in combinations(all_nodes, k + 1):
                if all(
                    path_lengths[u][v] <= self.distance_threshold
                    for u, v in combinations(combination, 2)
                ):
                    simplices[k - 2].add(tuple(sorted(combination)))

        for set_k_simplices in simplices:
            simplicial_complex.add_simplices_from(list(set_k_simplices))

        return self._get_lifted_topology(simplicial_complex, graph)
