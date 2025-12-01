"""This module implements the ModularityMaximizationLifting class."""

import torch
import torch_geometric

from topobench.transforms.liftings.graph2hypergraph.base import (
    Graph2HypergraphLifting,
)


class ModularityMaximizationLifting(Graph2HypergraphLifting):
    r"""Lifts graphs to hypergraph domain using modularity maximization and community detection.

    This method creates hyperedges based on the community structure of the graph and
    k-nearest neighbors within each community.

    Parameters
    ----------
    num_communities : int, optional
        The number of communities to detect. Default is 2.
    k_neighbors : int, optional
        The number of nearest neighbors to consider within each community. Default is 3.
    use_graph_connectivity : bool, optional
        If True include the original edges as hyperedges of dimensions 2. Default if False.
    **kwargs : optional
        Additional arguments for the base class.
    """

    def __init__(
        self,
        num_communities=2,
        k_neighbors=3,
        use_graph_connectivity=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_communities = num_communities
        self.k_neighbors = k_neighbors
        self.use_graph_connectivity = use_graph_connectivity

    def modularity_matrix(self, data):
        r"""Compute the modularity matrix B of the graph.

        B_ij = A_ij - (k_i * k_j) / (2m)

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input graph data.

        Returns
        -------
        torch.Tensor
            The modularity matrix B.
        """
        a = torch.zeros((data.num_nodes, data.num_nodes))
        a[data.edge_index[0], data.edge_index[1]] = 1
        k = a.sum(dim=1)
        m = data.edge_index.size(1) / 2
        return a - torch.outer(k, k) / (2 * m)

    def kmeans(self, x, n_clusters, n_iterations=100):
        r"""Perform k-means clustering on the input data.

        Note: This implementation uses random initialization, so results may vary
        between runs even for the same input data.

        Parameters
        ----------
        x : torch.Tensor
            The input data to cluster.
        n_clusters : int
            The number of clusters to form.
        n_iterations : int, optional
            The maximum number of iterations. Default is 100.

        Returns
        -------
        torch.Tensor
            The cluster assignments for each input point.

        Warning
        -------
        Due to random initialization of centroids, the resulting hyperedges
        may differ each time the code is run, even with the same input.
        """
        # Initialize cluster centers randomly
        centroids = x[
            torch.randperm(
                x.shape[0],
            )[:n_clusters]
        ]
        cluster_assignments = torch.zeros(x.shape[0], dtype=torch.long)
        for _ in range(n_iterations):
            # Assign points to the nearest centroid
            distances = torch.cdist(x, centroids)
            cluster_assignments = torch.argmin(distances, dim=1)

            # Update centroids
            new_centroids = torch.stack(
                [
                    x[cluster_assignments == k].mean(dim=0)
                    for k in range(n_clusters)
                ]
            )

            if torch.allclose(centroids, new_centroids):
                break

            centroids = new_centroids

        return cluster_assignments

    def detect_communities(self, b):
        r"""Detect communities using spectral clustering on the modularity matrix.

        Parameters
        ----------
        b : torch.Tensor
            The modularity matrix.

        Returns
        -------
        torch.Tensor
            The community assignments for each node.
        """
        eigvals, eigvecs = torch.linalg.eigh(b)
        leading_eigvecs = eigvecs[
            :, torch.argsort(eigvals, descending=True)[: self.num_communities]
        ]

        # Use implemented k-means clustering on the leading eigenvectors
        return self.kmeans(leading_eigvecs, self.num_communities)

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Lift the graph topology to a hypergraph based on community structure and k-nearest neighbors.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input graph data.

        Returns
        -------
        dict
            A dictionary containing the incidence matrix of the hypergraph, number of hyperedges,
            and the original node features.
        """
        b = self.modularity_matrix(data)
        community_assignments = self.detect_communities(b)

        num_nodes = data.x.shape[0]
        num_hyperedges = num_nodes
        incidence_matrix = torch.zeros(num_nodes, num_nodes)

        for i in range(num_nodes):
            # Find nodes in the same community
            same_community = (
                (community_assignments == community_assignments[i])
                .nonzero()
                .view(-1)
            )

            # Calculate distances to nodes in the same community
            distances = torch.norm(
                data.x[i].unsqueeze(0) - data.x[same_community], dim=1
            )

            # Select k nearest neighbors within the community
            k = min(self.k_neighbors, len(same_community))
            _, nearest_indices = torch.topk(distances, k, largest=False)
            nearest_neighbors = same_community[nearest_indices]

            # Create a hyperedge
            incidence_matrix[i, nearest_neighbors] = 1
            incidence_matrix[i, i] = 1  # Include the node itself

        incidence_matrix = incidence_matrix.to_sparse_coo()

        if self.use_graph_connectivity and data.edge_index is not None:
            # 1. Extract unique undirected edges to avoid duplicate hyperedges
            row, col = data.edge_index
            mask = row < col
            unique_edges = data.edge_index[:, mask]  # Shape [2, num_new_edges]

            num_new_edges = unique_edges.shape[1]

            if num_new_edges > 0:
                # 2. Prepare indices for the new sparse tensor
                new_row_indices = torch.cat([unique_edges[0], unique_edges[1]])

                # The column indices correspond to the new hyperedge IDs
                new_col_indices = torch.arange(
                    num_new_edges, device=data.x.device
                ).repeat_interleave(2)

                # Stack to create sparse indices [2, nnz]
                new_indices = torch.stack([new_row_indices, new_col_indices])

                # Values are all 1s
                new_values = torch.ones(
                    new_indices.shape[1], device=data.x.device
                )

                # 3. Concatenate with existing sparse incidence matrix
                old_indices = incidence_matrix.indices()
                old_values = incidence_matrix.values()
                old_shape = incidence_matrix.shape

                # Shift the new column indices by the number of existing hyperedges
                new_indices[1] += old_shape[1]

                final_indices = torch.cat([old_indices, new_indices], dim=1)
                final_values = torch.cat([old_values, new_values])

                # Update shape and create new sparse tensor
                final_shape = (old_shape[0], old_shape[1] + num_new_edges)

                incidence_matrix = torch.sparse_coo_tensor(
                    final_indices, final_values, final_shape
                ).coalesce()

                num_hyperedges += num_new_edges

        return {
            "incidence_hyperedges": incidence_matrix,
            "num_hyperedges": num_hyperedges,
            "x_0": data.x,
        }
