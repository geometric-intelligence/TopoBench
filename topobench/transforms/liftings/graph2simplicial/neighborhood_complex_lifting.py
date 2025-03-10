"""This module implements the NeighborhoodComplexLifting class, which lifts graphs to simplicial complexes."""

import random
from itertools import combinations
from typing import Any

from toponetx.classes import SimplicialComplex
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from tqdm import tqdm

from topobench.transforms.liftings.graph2simplicial.base import (
    Graph2SimplicialLifting,
)


class NeighborhoodComplexLifting(Graph2SimplicialLifting):
    """Lifts graphs to a simplicial complex domain by identifying the neighborhood complex as k-simplices.

    Parameters
    ----------
    max_simplices : int, optional
        The maximum number of simplices to be added to the simplicial complex for each node. Default is 50.
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, max_simplices=10, **kwargs):
        super().__init__(**kwargs)
        self.max_simplices = max_simplices

    def lift_topology(self, data: Data) -> dict:
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
        data.edge_index = to_undirected(data.edge_index)
        graph = self._generate_graph_from_data(data)
        simplicial_complex = SimplicialComplex(simplices=graph)
        simplices: list[set[tuple[Any, ...]]] = [
            set() for _ in range(2, self.complex_dim + 1)
        ]
        # For every node u
        disable = len(graph.nodes) < 500
        for u in tqdm(graph.nodes, desc="Adding simplices", disable=disable):
            neighbourhood_complex = set()
            neighbourhood_complex.add(u)
            first_neighbors = set(graph.neighbors(u))
            for v in first_neighbors:
                neighbourhood_complex.update(list(graph.neighbors(v)))
            neighbourhood_complex -= first_neighbors
            random.shuffle(list(neighbourhood_complex))
            for i in range(2, self.complex_dim + 1):
                for num_c, c in enumerate(
                    combinations(neighbourhood_complex, i + 1)
                ):
                    simplices[i - 2].add(tuple(c))
                    if num_c >= self.max_simplices:
                        break

        for set_k_simplices in simplices:
            simplicial_complex.add_simplices_from(list(set_k_simplices))

        feature_dict = {i: f for i, f in enumerate(data["x"])}

        simplicial_complex.set_simplex_attributes(
            feature_dict, name="features"
        )

        return self._get_lifted_topology(simplicial_complex, graph)
