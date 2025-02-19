import networkx as nx
import torch_geometric
from toponetx.classes import SimplicialComplex

from modules.transforms.liftings.graph2simplicial.base import (
    Graph2SimplicialLifting,
)


class SimplicialLineLifting(Graph2SimplicialLifting):
    r"""Lifts graphs to a simplicial complex domain by considering line simplicial complex.

    Line simplicial complex is a clique complex of the line graph. Line graph is a graph, in which
    the vertices are the edges in the initial graph, and two vertices are adjacent if the corresponding
    edges are adjacent in the initial graph.

    Parameters
    ----------
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Lifts the topology of a graph to simplicial domain via line simplicial complex construction.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        ----------
        dict
            The lifted topology.
        """

        graph = self._generate_graph_from_data(data)
        line_graph = nx.line_graph(graph)

        node_features = {
            node: ((data.x[node[0], :] + data.x[node[1], :]) / 2)
            for node in list(line_graph.nodes)
        }

        cliques = nx.find_cliques(line_graph)
        simplices = list(cliques)  # list(map(lambda x: set(x), cliques))

        # we need to rename simplices here since now vertices are named as pairs
        self.rename_vertices_dict = {
            node: i for i, node in enumerate(line_graph.nodes)
        }
        self.rename_vertices_dict_inverse = {
            i: node for node, i in self.rename_vertices_dict.items()
        }
        renamed_line_graph = nx.relabel_nodes(
            line_graph, self.rename_vertices_dict
        )

        renamed_simplices = [
            {self.rename_vertices_dict[vertex] for vertex in simplex}
            for simplex in simplices
        ]

        renamed_node_features = {
            self.rename_vertices_dict[node]: value
            for node, value in node_features.items()
        }

        simplicial_complex = SimplicialComplex(simplices=renamed_simplices)
        self.complex_dim = simplicial_complex.dim

        simplicial_complex.set_simplex_attributes(
            renamed_node_features, name="features"
        )

        return self._get_lifted_topology(
            simplicial_complex, renamed_line_graph
        )
