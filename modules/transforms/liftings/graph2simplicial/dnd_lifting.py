import random
from itertools import combinations

import networkx as nx
from toponetx.classes import SimplicialComplex
from torch_geometric.data import Data

from modules.transforms.liftings.graph2simplicial.base import Graph2SimplicialLifting


class SimplicialDnDLifting(Graph2SimplicialLifting):
    r"""Lifts graphs to simplicial complex domain using a Dungeons & Dragons inspired system.

    Parameters
    ----------
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def lift_topology(self, data: Data) -> dict:
        r"""Lifts the topology of a graph to a simplicial complex using Dungeons & Dragons (D&D) inspired mechanics.

        Parameters
        ----------
        data : Data
            The input data to be lifted.

        Returns
        -------
        dict
            The lifted topology.
        """
        graph = self._generate_graph_from_data(data)
        simplicial_complex = SimplicialComplex()

        characters = self._assign_attributes(graph)
        simplices = [set() for _ in range(2, self.complex_dim + 1)]

        for node in graph.nodes:
            simplicial_complex.add_node(node, features=data.x[node])

        for node in graph.nodes:
            character = characters[node]
            for k in range(1, self.complex_dim):
                dice_roll = self._roll_dice(character, k)
                neighborhood = list(
                    nx.single_source_shortest_path_length(
                        graph, node, cutoff=dice_roll
                    ).keys()
                )
                for combination in combinations(neighborhood, k + 1):
                    simplices[k - 1].add(tuple(sorted(combination)))

        for set_k_simplices in simplices:
            simplicial_complex.add_simplices_from(list(set_k_simplices))

        return self._get_lifted_topology(simplicial_complex, graph)

    def _assign_attributes(self, graph):
        """Assign D&D-inspired attributes based on node properties."""
        degrees = nx.degree_centrality(graph)
        clustering = nx.clustering(graph)
        closeness = nx.closeness_centrality(graph)
        eigenvector = nx.eigenvector_centrality(graph)
        betweenness = nx.betweenness_centrality(graph)
        pagerank = nx.pagerank(graph)

        attributes = {}
        for node in graph.nodes:
            attributes[node] = {
                "Degree": degrees[node],
                "Clustering": clustering[node],
                "Closeness": closeness[node],
                "Eigenvector": eigenvector[node],
                "Betweenness": betweenness[node],
                "Pagerank": pagerank[node],
            }
        return attributes

    def _roll_dice(self, attributes, k):
        """Simulate a D20 dice roll influenced by node attributes where a different attribute is used based on the simplex level."""

        attribute = None
        if k == 1:
            attribute = attributes["Degree"]
        elif k == 2:
            attribute = attributes["Clustering"]
        elif k == 3:
            attribute = attributes["Closeness"]
        elif k == 4:
            attribute = attributes["Eigenvector"]
        elif k == 5:
            attribute = attributes["Betweenness"]
        else:
            attribute = attributes["Pagerank"]

        base_roll = random.randint(1, 20)
        modifier = int(attribute * 20)
        return base_roll + modifier
