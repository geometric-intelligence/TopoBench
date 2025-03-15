"""Mixture of Gaussians and Minimum Spanning Tree (MoGMST) Lifting."""

import numpy as np
import torch
import torch_geometric
from networkx import from_numpy_array, minimum_spanning_tree
from sklearn.metrics import pairwise_distances
from sklearn.mixture import GaussianMixture

from topobench.transforms.liftings.pointcloud2hypergraph.base import (
    PointCloud2HypergraphLifting,
)


class MoGMSTLifting(PointCloud2HypergraphLifting):
    r"""Lift a point cloud to a hypergraph.

    We find a Mixture of Gaussians and then create a Minimum Spanning Tree (MST) between the means of the Gaussians.

    Parameters
    ----------
    min_components : int or None, optional
        The minimum number of components for the Mixture of Gaussians model.
        It needs to be at least 1 (default: None).
    max_components : int or None, optional
        The maximum number of components for the Mixture of Gaussians model.
        It needs to be greater or equal than min_components (default: None).
    random_state : int, optional
        The random state for the Mixture of Gaussians model (default: None).
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(
        self,
        min_components=None,
        max_components=None,
        random_state=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.min_components = min_components
        self.max_components = max_components
        self.random_state = random_state

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        """Lift the topology of a graph to a hypergraph.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        dict
            The lifted topology.
        """
        # Find a mix of Gaussians
        number_of_points = data.x.shape[0]
        labels, num_components, means = self.find_mog(data.x.numpy())

        # If no labels are found, return a single hyperedge with all the nodes
        if labels is None:
            incidence = torch.ones((number_of_points, 1))
            incidence = incidence.to_sparse_coo()
            return {
                "incidence_hyperedges": incidence,
                "num_hyperedges": 1,
                "x_0": data.x,
            }
        # Create MST
        distance_matrix = pairwise_distances(means)
        original_graph = from_numpy_array(distance_matrix)
        mst = minimum_spanning_tree(original_graph)

        # Create hypergraph incidence
        incidence = torch.zeros((number_of_points, 2 * num_components))

        # Add to which Gaussian the points belong to
        nodes = torch.arange(0, number_of_points, dtype=torch.int32)
        lbls = torch.tensor(labels, dtype=torch.int32)
        values = torch.ones(number_of_points)
        incidence[nodes, lbls] = values

        # Add neighbours in MST
        for i, j in mst.edges():
            mask_i = labels == i
            mask_j = labels == j
            incidence[mask_i, num_components + j] = 1
            incidence[mask_j, num_components + i] = 1

        incidence = incidence.clone().detach().to_sparse_coo()
        return {
            "incidence_hyperedges": incidence,
            "num_hyperedges": 2 * num_components,
            "x_0": data.x,
        }

    def find_mog(self, data) -> tuple[np.ndarray, int, np.ndarray]:
        """Find the best number of components for a Mixture of Gaussians model.

        Parameters
        ----------
        data : np.ndarray
            The input data to be fitted.

        Returns
        -------
        tuple[np.ndarray, int, np.ndarray]
            The labels of the data, the number of components and the means of the components.
        """
        possible_num_components = [
            self.min_components if self.min_components is not None else 1
        ]
        if self.min_components is not None and self.max_components is not None:
            possible_num_components = range(
                self.min_components, self.max_components + 1
            )
        elif self.min_components is None and self.max_components is None:
            possible_num_components = [
                2**i for i in range(1, int(np.log2(data.shape[0] / 2)) + 1)
            ]

        best_score = float("inf")
        best_labels = None
        best_num_components = 0
        means = None
        for i in possible_num_components:
            gm = GaussianMixture(
                n_components=i, random_state=self.random_state
            )
            labels = gm.fit_predict(data)
            score = gm.aic(data)
            if score < best_score:
                best_score = score
                best_labels = labels
                best_num_components = i
                means = gm.means_
        return best_labels, best_num_components, means
