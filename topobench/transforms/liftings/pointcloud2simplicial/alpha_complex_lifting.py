"""This module defines the AlphaComplexLifting class, which lifts point clouds to simplicial complexes using the alpha complex method."""

import gudhi
import torch_geometric
from toponetx.classes import SimplicialComplex

from topobench.transforms.liftings.pointcloud2simplicial.base import (
    PointCloud2SimplicialLifting,
)


class AlphaComplexLifting(PointCloud2SimplicialLifting):
    r"""Lift point clouds to simplicial complex domain.

    The lifting is done by generating the alpha complex using the Gudhi library. The alpha complex is a simplicial complex constructed from the finite cells of a Delaunay Triangulation. It has the same persistent homology as the Čech complex and is significantly smaller. When the alpha parameter is set to -1, the alpha complex is the Delaunay Triangulation.

    Parameters
    ----------
    alpha : float
        The alpha parameter of the alpha complex.
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, alpha: float, **kwargs):
        self.alpha = alpha
        super().__init__(**kwargs)

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Lift the topology of a point cloud to the alpha complex.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        dict
            The lifted topology.
        """
        ac = gudhi.AlphaComplex(data.x)
        stree = ac.create_simplex_tree()
        if self.alpha > 0:
            stree.prune_above_filtration(self.alpha)
        stree.prune_above_dimension(self.complex_dim)

        sc = SimplicialComplex(
            s for s, filtration_value in stree.get_simplices()
        )
        # If some points are excluded by the Alpha complex we need to keep them
        n_points = data.x.shape[0]
        if stree.num_vertices() != n_points:
            kept_indices = {simplex[0][0] for simplex in stree.get_skeleton(0)}
            all_indices = set(range(n_points))
            excluded_indices = list(all_indices - kept_indices)
            for excluded_index in excluded_indices:
                sc.add_node(excluded_index)

        lifted_topolgy = self._get_lifted_topology(sc)
        lifted_topolgy["x_0"] = data.x
        return lifted_topolgy
