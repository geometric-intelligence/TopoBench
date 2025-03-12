"""This module implements the SimplicialVietorisRipsLifting class, which lifts graphs to simplicial complexes."""

import random

import networkx as nx
import torch_geometric
from toponetx.classes import SimplicialComplex
from tqdm import tqdm

from topobench.transforms.liftings.graph2simplicial.base import (
    Graph2SimplicialLifting,
)


class SimplicialVietorisRipsLifting(Graph2SimplicialLifting):
    r"""Lift graphs to simplicial complex domain using the Vietoris-Rips complex based on pairwise distances.

    Parameters
    ----------
    distance_threshold : float
        The maximum distance between vertices to form a simplex.
    max_simplices : int
        Max number of simplices to create for a given node and rank.
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, distance_threshold=1.0, max_simplices=5, **kwargs):
        super().__init__(**kwargs)
        self.distance_threshold = distance_threshold
        self.max_simplices = max_simplices

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Lift topology of a graph to a simplicial complex using the Vietoris-Rips complex.

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
            disable = len(all_nodes) < 1000
            for node in tqdm(
                all_nodes,
                desc=f"Adding simplices of rank {k}",
                disable=disable,
            ):
                added_simplices = 0
                connected_nodes = [
                    n
                    for n in list(path_lengths[node].keys())[1:]
                    if path_lengths[node][n] <= self.distance_threshold
                ]
                random.shuffle(connected_nodes)
                for current_idx in range(len(connected_nodes)):
                    simplex = nodes_search(
                        [node],
                        max_num_nodes=k + 1,
                        current_idx=current_idx,
                        path_lengths=path_lengths,
                        intersection=connected_nodes,
                        distance_threshold=self.distance_threshold,
                    )
                    if simplex is not None:
                        simplices[k - 2].add(tuple(sorted(simplex)))
                        added_simplices += 1
                        if added_simplices >= self.max_simplices:
                            break

        for set_k_simplices in simplices:
            simplicial_complex.add_simplices_from(list(set_k_simplices))

        return self._get_lifted_topology(simplicial_complex, graph)


def nodes_search(
    nodes,
    max_num_nodes,
    current_idx,
    path_lengths,
    intersection,
    distance_threshold,
):
    """Find nodes that all have distance less than distance_threshold.

    Parameters
    ----------
    nodes : list
        Current list of nodes.
    max_num_nodes : int
        Target number of nodes to return.
    current_idx : int
        Index of the node from intersection to add. Can be None.
    path_lengths : dict
        Dictionary with path lengths between nodes in the graph.
    intersection : list
        List of possible nodes to add.
    distance_threshold : int
        Max distance to consider for nodes.

    Returns
    -------
    list
        List of nodes where each pair of nodes had distance less or equal to the threshold. Returns None if there was no set of nodes satisfying the condition.
    """
    if len(nodes) == max_num_nodes:
        return nodes
    elif len(intersection) == 0:
        return None

    new_node = None
    if current_idx is not None:
        new_node = intersection[current_idx]
    else:
        for node in intersection:
            if node not in nodes:
                new_node = node
                break
    nodes.append(new_node)
    new_connected_nodes = set(
        [
            n
            for n in list(path_lengths[new_node].keys())[1:]
            if path_lengths[new_node][n] <= distance_threshold
        ]
    )
    intersection = list(set(intersection) & set(new_connected_nodes))
    random.shuffle(intersection)
    return nodes_search(
        nodes,
        max_num_nodes,
        None,
        path_lengths,
        intersection,
        distance_threshold,
    )
