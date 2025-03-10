"""Simulates a Dungeons & Dragons inspired system to lift graphs to simplicial complexes."""

import random
from itertools import combinations

import networkx as nx
from toponetx.classes import SimplicialComplex
from torch_geometric.data import Data

from topobench.transforms.liftings.graph2simplicial.base import (
    Graph2SimplicialLifting,
)


class SimplicialDnDLifting(Graph2SimplicialLifting):
    r"""Lift graphs to simplicial complex domain using a Dungeons & Dragons inspired system.

    Parameters
    ----------
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def lift_topology(self, data: Data, max_simplices=10) -> dict:
        r"""Lift the topology of a graph to a simplicial complex using Dungeons & Dragons (D&D) inspired mechanics.

        Parameters
        ----------
        data : Data
            The input data to be lifted.
        max_simplices : int
            Maximum number of simplices to add for each node attribute.

        Returns
        -------
        dict
            The lifted topology.
        """
        graph = self._generate_graph_from_data(data)
        simplicial_complex = SimplicialComplex()

        characters = self._assign_attributes(graph)
        simplices = [set() for _ in range(1, self.complex_dim + 1)]

        for node in graph.nodes:
            simplicial_complex.add_node(node, features=data.x[node])

        for node in graph.nodes:
            character = characters[node]
            for k in range(1, self.complex_dim + 1):
                dice_roll = self._roll_dice(character, k)
                neighborhood = list(
                    nx.single_source_shortest_path_length(
                        graph, node, cutoff=dice_roll
                    ).keys()
                )
                neighborhood = neighborhood[1:]
                random.shuffle(neighborhood)
                neighborhood = [node, *neighborhood]
                for i, combination in enumerate(
                    combinations(neighborhood, k + 1)
                ):
                    simplices[k - 1].add(tuple(sorted(combination)))
                    if i >= max_simplices:
                        break

        for set_k_simplices in simplices:
            simplicial_complex.add_simplices_from(list(set_k_simplices))

        return self._get_lifted_topology(simplicial_complex, graph)

    def _assign_attributes(self, graph):
        """Assign D&D-inspired attributes based on node properties.

        Parameters
        ----------
        graph : nx.Graph
            The input graph.

        Returns
        -------
        dict
            The assigned attributes.
        """
        degrees = nx.degree_centrality(graph)
        clustering = nx.clustering(graph)
        # closeness = nx.closeness_centrality(graph)
        eigenvector = nx.eigenvector_centrality(graph, tol=1e-3)
        # betweenness = nx.betweenness_centrality(graph)
        pagerank = nx.pagerank(graph)

        attributes = {}
        for node in graph.nodes:
            attributes[node] = {
                "Degree": degrees[node],
                "Clustering": clustering[node],
                # "Closeness": closeness[node],
                "Eigenvector": eigenvector[node],
                # "Betweenness": betweenness[node],
                "Pagerank": pagerank[node],
            }
        return attributes

    def _roll_dice(self, attributes, k):
        """Simulate a D20 dice roll influenced by node attributes where a different attribute is used based on the simplex level.

        Parameters
        ----------
        attributes : dict
            The attributes of the node.
        k : int
            The level of the simplex.

        Returns
        -------
        int
            The dice roll.
        """

        attribute = None
        if k == 1:
            attribute = attributes["Degree"]
        elif k == 2:
            attribute = attributes["Clustering"]
        # elif k == 3:
        #     attribute = attributes["Closeness"]
        elif k == 3:
            attribute = attributes["Eigenvector"]
        # elif k == 5:
        #     attribute = attributes["Betweenness"]
        else:
            attribute = attributes["Pagerank"]

        base_roll = random.randint(1, 20)
        modifier = int(attribute * 20)
        return (
            base_roll + modifier
        ) / 10  # 4 seems like a reasonable max number of hops to consider
