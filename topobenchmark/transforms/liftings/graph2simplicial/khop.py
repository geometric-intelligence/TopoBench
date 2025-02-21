"""This module implements the k-hop lifting of graphs to simplicial complexes."""

import random
from itertools import combinations
from typing import Any

import torch_geometric
from toponetx.classes import SimplicialComplex

from topobenchmark.transforms.liftings.graph2simplicial.base import (
    Graph2SimplicialLifting,
)


class SimplicialKHopLifting(Graph2SimplicialLifting):
    r"""Lift graphs to simplicial complex domain.

    The function lifts a graph to a simplicial complex by considering k-hop
    neighborhoods. For each node its neighborhood is selected and then all the
    possible simplices, when considering the neighborhood as a clique, are
    added to the simplicial complex. For this reason this lifting does not
    conserve the initial graph topology.

    Parameters
    ----------
    max_k_simplices : int, optional
        The maximum number of k-simplices to consider. Default is 5000.
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, max_k_simplices=5000, **kwargs):
        super().__init__(**kwargs)
        self.max_k_simplices = max_k_simplices

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(max_k_simplices={self.max_k_simplices!r})"

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Lift the topology to simplicial complex domain.

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
        edge_index = torch_geometric.utils.to_undirected(data.edge_index)
        simplices: list[set[tuple[Any, ...]]] = [
            set() for _ in range(2, self.complex_dim + 1)
        ]
        for n in range(graph.number_of_nodes()):
            # Find 1-hop node n neighbors
            neighbors, _, _, _ = torch_geometric.utils.k_hop_subgraph(
                n, 1, edge_index
            )
            if n not in neighbors:
                neighbors.append(n)
            neighbors = neighbors.numpy()
            neighbors = set(neighbors)
            for i in range(1, self.complex_dim):
                for c in combinations(neighbors, i + 1):
                    simplices[i - 1].add(tuple(c))
        for set_k_simplices in simplices:
            list_k_simplices = list(set_k_simplices)
            if len(set_k_simplices) > self.max_k_simplices:
                random.shuffle(list_k_simplices)
                list_k_simplices = list_k_simplices[: self.max_k_simplices]
            simplicial_complex.add_simplices_from(list_k_simplices)
        return self._get_lifted_topology(simplicial_complex, graph)
