"""Generic triangle classification utilities for graph analysis.

This module provides a base class for extracting and classifying triangles
in weighted graphs using efficient algorithms.
"""

import networkx as nx
import torch


class TriangleClassifier:
    """Base class for extracting and classifying triangles in graphs.

    Provides generic triangle enumeration and edge weight extraction.
    Subclasses should override _classify_role() for domain-specific role definitions.

    Parameters
    ----------
    min_weight : float, optional
        Minimum edge weight to consider, by default 0.2.
    """

    def __init__(self, min_weight: float = 0.2):
        """Initialize triangle classifier.

        Parameters
        ----------
        min_weight : float, optional
            Minimum edge weight to consider as valid edge, by default 0.2
        """
        self.min_weight = min_weight

    def enumerate_triangles(self, G: nx.Graph) -> list:
        """Enumerate all triangles in a graph using efficient O(n^3) enumeration.

        Parameters
        ----------
        G : nx.Graph
            NetworkX graph object with edges and optional weights.

        Returns
        -------
        list of tuple
            Each tuple is (a, b, c) representing a triangle.
        """
        triangles = []
        nodes = list(G.nodes())
        for i, a in enumerate(nodes):
            neighbors_a = set(G.neighbors(a))
            for j, b in enumerate(nodes[i + 1 :], start=i + 1):
                if b not in neighbors_a:
                    continue
                neighbors_b = set(G.neighbors(b))
                for c in nodes[j + 1 :]:
                    if c in neighbors_a and c in neighbors_b:
                        # Found triangle (a,b,c)
                        triangles.append((a, b, c))  # noqa: PERF401 (conditional append, not extend)

        return triangles

    def classify_and_weight_triangles(
        self, triangles: list, G: nx.Graph
    ) -> list:
        """Classify triangles and add edge weights and role information.

        Parameters
        ----------
        triangles : list of tuple
            List of triangles, each as (a, b, c) node indices.
        G : nx.Graph
            NetworkX graph with edge weights and adjacency information.

        Returns
        -------
        list of dict
            Each dict contains {'nodes': (a,b,c), 'edge_weights': [w1,w2,w3], 'role': str, 'label': int}.
        """
        triangle_data = []
        for nodes in triangles:
            a, b, c = nodes

            # Get edge weights
            w_ab = G[a][b].get("weight", self.min_weight)
            w_bc = G[b][c].get("weight", self.min_weight)
            w_ac = G[a][c].get("weight", self.min_weight)
            edge_weights_tri = [w_ab, w_bc, w_ac]

            # Classify role (subclasses override _classify_role for domain-specific logic)
            role = self._classify_role(G, nodes, edge_weights_tri)

            triangle_data.append(
                {
                    "nodes": nodes,
                    "edge_weights": edge_weights_tri,
                    "role": role,
                    "label": self._role_to_label(role),
                }
            )

        return triangle_data

    def extract_triangles(
        self,
        edge_index: torch.Tensor,
        edge_weights: torch.Tensor,
        num_nodes: int,
    ) -> list:
        """Extract all triangles from graph (convenience method).

        Combines enumerate_triangles and classify_and_weight_triangles.

        Parameters
        ----------
        edge_index : torch.Tensor
            Edge connectivity, shape (2, num_edges).
        edge_weights : torch.Tensor
            Edge weights, shape (num_edges,).
        num_nodes : int
            Number of nodes.

        Returns
        -------
        list of dict
            Each dict contains {'nodes': (a,b,c), 'edge_weights': [w1,w2,w3], 'role': str, 'label': int}.
        """
        # Build networkx graph
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))

        for i in range(edge_index.shape[1]):
            u = edge_index[0, i].item()
            v = edge_index[1, i].item()
            w = edge_weights[i].item()
            G.add_edge(u, v, weight=w)

        # Enumerate and classify triangles
        triangles = self.enumerate_triangles(G)
        triangle_data = self.classify_and_weight_triangles(triangles, G)

        return triangle_data

    def _classify_role(
        self, G: nx.Graph, nodes: tuple, edge_weights: list
    ) -> str:
        """Classify role of triangle based on edge weights and embedding.

        This method should be overridden in subclasses to provide domain-specific
        role classification logic.

        Parameters
        ----------
        G : nx.Graph
            The graph.
        nodes : tuple
            Three node indices forming the triangle.
        edge_weights : list
            Three edge weights.

        Returns
        -------
        str
            Role string describing the triangle's role.

        Raises
        ------
        NotImplementedError
            If not overridden in subclass.
        """
        raise NotImplementedError(
            "Subclasses must override _classify_role() to define domain-specific role classification"
        )

    def _role_to_label(self, role_str: str) -> int:
        """Convert role string to integer label.

        This method should be overridden in subclasses to define the mapping
        from role strings to numeric labels.

        Parameters
        ----------
        role_str : str
            Role string (e.g., "core_strong").

        Returns
        -------
        int
            Numeric label.

        Raises
        ------
        NotImplementedError
            If not overridden in subclass.
        """
        raise NotImplementedError(
            "Subclasses must override _role_to_label() to define role-to-label mapping"
        )
